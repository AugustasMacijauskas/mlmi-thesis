#!/bin/bash

# Example string
my_string="Hello, world!"
my_string=""

# Check if the string is not empty using -n operator
if [[ -n "$my_string" ]]; then
  echo "The string is not empty"
else
  echo "The string is empty"
fi
