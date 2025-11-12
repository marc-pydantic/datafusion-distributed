#[cfg(all(feature = "integration", test))]
mod tests {
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::physical_plan::{displayable, execute_stream};
    use datafusion_distributed::test_utils::localhost::start_localhost_context;
    use datafusion_distributed::{
        DefaultSessionBuilder, DistributedConfig, apply_network_boundaries, assert_snapshot,
        distribute_plan,
    };
    use futures::TryStreamExt;
    use std::error::Error;

    #[tokio::test]
    async fn test_left_join_empty_right_side_bug() -> Result<(), Box<dyn Error>> {
        let (ctx, _guard) = start_localhost_context(4, DefaultSessionBuilder).await;

        // Query that does a left join, should return exactly one row.
        let df = ctx
            .sql(
                r#"
                SELECT value
                FROM generate_series(1, 1)
                LEFT JOIN (SELECT 1 as join_key WHERE false) ON value = join_key
            "#,
            )
            .await?;

        let physical = df.create_physical_plan().await?;
        let physical_str = displayable(physical.as_ref()).indent(true).to_string();

        let cfg = DistributedConfig::default().with_network_shuffle_tasks(4);
        let physical_distributed = apply_network_boundaries(physical.clone(), &cfg)?;
        let physical_distributed = distribute_plan(physical_distributed)?;

        let mut stream = execute_stream(physical_distributed, ctx.task_ctx())?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            results.push(batch);
        }

        // Count total rows
        let total_rows: usize = results.iter().map(|batch| batch.num_rows()).sum();

        // Expected: 1 row (from left side with NULL from empty right side)
        // Bug: Multiple rows due to distributed execution multiplying by number of nodes

        assert_snapshot!(physical_str, @r"
        CoalesceBatchesExec: target_batch_size=8192
          HashJoinExec: mode=Partitioned, join_type=Left, on=[(value@0, join_key@0)], projection=[value@0]
            CoalesceBatchesExec: target_batch_size=8192
              RepartitionExec: partitioning=Hash([value@0], 3), input_partitions=3
                RepartitionExec: partitioning=RoundRobinBatch(3), input_partitions=1
                  LazyMemoryExec: partitions=1, batch_generators=[generate_series: start=1, end=1, batch_size=8192]
            CoalesceBatchesExec: target_batch_size=8192
              RepartitionExec: partitioning=Hash([join_key@0], 3), input_partitions=1
                EmptyExec
        ");

        let formatted_batches = pretty_format_batches(&results)?;

        assert_snapshot!(formatted_batches, @r"
        +-------+
        | value |
        +-------+
        | 1     |
        +-------+
        ");
        assert_eq!(total_rows, 1);

        Ok(())
    }
}
