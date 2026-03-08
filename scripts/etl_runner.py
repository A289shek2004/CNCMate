import time
import subprocess

print("Starting ETL auto-runner")

while True:

    print("Running ETL pipeline...")

    subprocess.run(["python","notebooks/01_etl_pipeline.py"])

    time.sleep(10)