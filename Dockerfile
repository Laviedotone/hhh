FROM python:3.10-slim

WORKDIR /home

# Copy the requirements file inside of the req folder first to leverage Docker's layer caching
# This step only re-runs if requirements.txt changes.
COPY req /home

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project code (calculator.py, test_calculator.py, etc.)
# into the /home directory in the container. 
# Do you really need to do this?????? ;) maybe for local testing

COPY . .
