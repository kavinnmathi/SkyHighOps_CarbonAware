-- Starter query for the OPTIONAL real workload extract (Google ClusterData2019, cell "a").
-- IMPORTANT before running:
--   1. Verify the table path and column names against the official documentation
--      ("Google cluster-usage traces v3", linked from https://github.com/google/cluster-data).
--   2. Look at the "This query will process X" estimate in the BigQuery console first.
--   3. Trace times are in microseconds; the window below is the first 7 days after the trace offset.
--   4. Export the result as CSV and save it as data/real/google_trace_extract.csv

SELECT
  collection_id,
  instance_index,
  MIN(start_time)             AS start_time,
  MAX(end_time)               AS end_time,
  AVG(average_usage.cpus)     AS avg_cpus,
  AVG(average_usage.memory)   AS avg_memory
FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
WHERE start_time BETWEEN 600000000 AND 604800000000   -- ~7 days, in microseconds
GROUP BY collection_id, instance_index
LIMIT 5000;
