"""
Snakefile for running the sequenza-pipeline command for all patients

This Snakefile will:
1. Detect all patient directories under the 'data' directory
2. Run the sequenza-pipeline command for each patient
3. Use the Docker container

Usage:
    snakemake --cores <cores>
"""
import os
from pathlib import Path

# Find all patient directories under the data directory
DATA_DIR = "data"
patient_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
PATIENTS = patient_dirs

# Output directory
OUTPUT_DIR = "results"

# Path to reference genome
REFERENCE_GENOME = config.get("reference_genome", "/input/reference/genome.fa.gz")

# Default target rule
rule all:
    input:
        expand("{output_dir}/{patient}/sequenza_results.done", 
               output_dir=OUTPUT_DIR, 
               patient=PATIENTS)

# Run sequenza-pipeline for each patient
rule run_sequenza:
    input:
        normal_bam = lambda wildcards: f"{DATA_DIR}/{wildcards.patient}/{wildcards.patient}_normal.bam",
        tumor_bam = lambda wildcards: f"{DATA_DIR}/{wildcards.patient}/{wildcards.patient}_tumor.bam"
    output:
        done = "{output_dir}/{patient}/sequenza_results.done"
    params:
        sample_id = lambda wildcards: wildcards.patient,
        outdir = lambda wildcards: f"{OUTPUT_DIR}/{wildcards.patient}"
    shell:
        """
        # Create output directory
        mkdir -p {params.outdir}
        
        # Run sequenza-pipeline command
        sequenza-pipeline \
            --sample-id {params.sample_id} \
            --normal-bam {input.normal_bam} \
            --tumor-bam {input.tumor_bam} \
            --reference-gz {REFERENCE_GENOME} \
            --out-dir {params.outdir}
        
        # Create a done file to mark completion
        touch {output.done}
        """