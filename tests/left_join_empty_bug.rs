#[cfg(all(feature = "integration", test))]
mod tests {
    use datafusion::arrow::util::pretty::pretty_format_batches;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::physical_plan::{displayable, execute_stream};
    use datafusion::prelude::SessionConfig;
    use datafusion_distributed::test_utils::in_memory_channel_resolver::InMemoryChannelResolver;
    use datafusion_distributed::test_utils::localhost::start_localhost_context;
    use datafusion_distributed::{
        DistributedExt, DistributedPhysicalOptimizerRule, assert_snapshot,
    };
    use futures::TryStreamExt;
    use std::error::Error;
    use std::sync::Arc;

    async fn build_distributed_state(
        ctx: datafusion_distributed::DistributedSessionBuilderContext,
    ) -> Result<datafusion::execution::SessionState, datafusion::error::DataFusionError> {
        let config = SessionConfig::new().with_target_partitions(4);

        let state = SessionStateBuilder::new()
            .with_runtime_env(ctx.runtime_env)
            .with_default_features()
            .with_distributed_channel_resolver(InMemoryChannelResolver::new(4))
            .with_physical_optimizer_rule(Arc::new(DistributedPhysicalOptimizerRule))
            .with_config(config)
            .build();

        Ok(state)
    }

    #[tokio::test]
    async fn test_left_join_empty_right_side_bug() -> Result<(), Box<dyn Error>> {
        let (ctx, _guard) = start_localhost_context(4, build_distributed_state).await;

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

        let mut stream = execute_stream(physical.clone(), ctx.task_ctx())?;
        let mut results = Vec::new();
        while let Some(batch) = stream.try_next().await? {
            results.push(batch);
        }

        // Count total rows
        let total_rows: usize = results.iter().map(|batch| batch.num_rows()).sum();

        // Expected: 1 row (from left side with NULL from empty right side)
        // Bug: Multiple rows due to distributed execution multiplying by number of nodes

        println!("Test result: {} rows", total_rows);
        if !results.is_empty() {
            println!("{}", pretty_format_batches(&results)?);
        }

        println!("Physical plan:");
        println!("{}", physical_str);

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
