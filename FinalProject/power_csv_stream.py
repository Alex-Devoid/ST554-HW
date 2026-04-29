#!/usr/bin/env python3

import time
from pathlib import Path

import pandas as pd


STREAM_SOURCE = "/Users/alexdevoid/Documents/Stats/ST554-HW/FinalProject/power_streaming_data.csv"
STREAM_OUTPUT_DIR = Path("/Users/alexdevoid/Documents/Stats/ST554-HW/FinalProject/power_stream_input")


# read the streaming data into a pandas dataframe
stream_pdf = pd.read_csv(STREAM_SOURCE)

# write 20 CSV batches of five sampled rows
for i in range(1, 21):
    sample_pdf = stream_pdf.sample(
        n=5,
        replace=False,
        random_state=123 + i,
    )
    output_path = STREAM_OUTPUT_DIR / f"chunk_{i}.csv"

    # write the batch without the pandas index
    sample_pdf.to_csv(output_path, index=False)
    print(f"batch {i} wrote {output_path.name}", flush=True)

    # pause between batches
    if i < 20:
        time.sleep(10)
