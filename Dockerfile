# 第一阶段：导出依赖
FROM python:3.10 AS requirements-stage
WORKDIR /tmp

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip config set global.trusted-host mirrors.aliyun.com

RUN pip install poetry poetry-plugin-export

COPY ./pyproject.toml /tmp/

RUN poetry lock
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# 第二阶段：正式运行环境
FROM python:3.10-slim
WORKDIR /code

COPY --from=requirements-stage /tmp/requirements.txt /code/requirements.txt
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com \
    && pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app
COPY ./frontend /code/frontend
COPY .env /code/.env

COPY start.sh /code/start.sh
RUN chmod +x /code/start.sh

EXPOSE 8000 8001

CMD ["/code/start.sh"]