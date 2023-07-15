#!/bin/bash

# Folder path containing example files
folder_path="."

# Number of mpi processes
num_processes=$1

if [ "$num_processes" -lt 2 ]; then
  echo "Number of processes should be greater than or equal to 2, $num_processes < 2"
else
  for file in "$folder_path"/*.py; do
      echo "Running $file with $num_processes processes"
      mpiexec -n "$num_processes" python "$file"
  done
fi
