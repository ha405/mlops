FROM public.ecr.aws/lambda/python:3.12

# Copy requirements and install
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Copy all files
COPY . ${LAMBDA_TASK_ROOT}

# The Lambda handler (file_name.handler_name)
CMD ["app.serve.handler"]