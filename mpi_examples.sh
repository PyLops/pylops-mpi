#!/bin/bash

# Folder path containing example files
cd $1 || exit;
folder_path="."

# Number of mpi processes
num_processes=$2

if [ "$num_processes" -ge 2 ]; then
  for file in "$folder_path"/*.py; do
      echo "Running $file with $num_processes processes"
      mpiexec -n "$num_processes" python "$file"
  done
else
  echo "Number of processes should be greater than or equal to 2, $num_processes < 2"
fi
