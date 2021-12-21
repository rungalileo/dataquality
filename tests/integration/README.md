## Integration Tests

Our integration tests are meant to test the dataquality client at some version against
the API at some version (currently main-main).

### Schedule
Integration tests run every day at midnight (`cron: '0 0 * * *'`)

### Contribute
To add an integration test, simply add a python script into the `tests/integration`
folder. Every python script in there is run during the test.

### Setup
These tests run against a local (docker) instance of postgres, minio, and the api.
See our [first](./tests/integration/mock_training_run.py) test for an example of
how to connect to the API server and Minio

The following environment variables are set for you
```
GALILEO_API_URL: "http://localhost:8088"
GALILEO_MINIO_URL: "127.0.0.1:9000"
GALILEO_MINIO_ACCESS_KEY: "minioadmin"
GALILEO_MINIO_SECRET_KEY: "minioadmin"
```
