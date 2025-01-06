FROM curlimages/curl:8.11.1

USER root
RUN apk add --no-cache bash

COPY src/launch_glonet_daily_forecast_process.sh /tmp/launch_glonet_daily_forecast_process.sh

CMD [ "sh", "-c", "/tmp/launch_glonet_daily_forecast_process.sh" ]
