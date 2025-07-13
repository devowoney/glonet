#!/usr/bin/env bash

read_dom () {
    local IFS=\>
    read -d \< ENTITY CONTENT
}

curlKeycloakCommand="curl --silent -X POST https://auth.dive.edito.eu/auth/realms/datalab/protocol/openid-connect/token -H 'Content-Type: application/x-www-form-urlencoded' -d 'client_id=onyxia-minio' -d 'grant_type=refresh_token' -d 'refresh_token=$EDITO_MINIO_OFFLINE_TOKEN' -d 'scope=openid+email+profile'"
curlKeycloakResult=$(eval $curlKeycloakCommand)
keycloakToken=$(echo "$curlKeycloakResult" | tr ',' '\n' | grep -o '"access_token":"[^"]*' | sed 's/"access_token":"//')

AWS_S3_ENDPOINT=${AWS_S3_ENDPOINT="minio.dive.edito.eu"}
S3_ENDPOINT=${S3_ENDPOINT="https://$AWS_S3_ENDPOINT"}

curlMinioCommand="curl --silent -X POST '$S3_ENDPOINT?Action=AssumeRoleWithWebIdentity&WebIdentityToken=$keycloakToken&DurationSeconds=86400&Version=2011-06-15'"
while read_dom; do
    if [[ $ENTITY = "AccessKeyId" ]]; then
        AWS_ACCESS_KEY_ID=$CONTENT
    elif [[ $ENTITY = "SecretAccessKey" ]]; then
        AWS_SECRET_ACCESS_KEY=$CONTENT
    elif [[ $ENTITY = "SessionToken" ]]; then
        AWS_SESSION_TOKEN=$CONTENT
    fi
done < <(eval $curlMinioCommand)

curlKeycloakCommand="curl --silent -X POST 'https://auth.dive.edito.eu/auth/realms/datalab/protocol/openid-connect/token' -H 'Content-Type: application/x-www-form-urlencoded' -d 'client_id=edito' -d 'grant_type=refresh_token' -d 'refresh_token=$EDITO_OFFLINE_TOKEN' -d 'scope=openid'"
curlKeycloakResult=$(eval $curlKeycloakCommand)
keycloakToken=$(echo "$curlKeycloakResult" | tr ',' '\n' | grep -o '"access_token":"[^"]*' | sed 's/"access_token":"//')

EDITO_ACCESS_TOKEN=$keycloakToken

curl --silent -X DELETE \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $EDITO_ACCESS_TOKEN" \
  -H "ONYXIA-REGION: WAW3-1" \
  -H "ONYXIA-PROJECT: project-glonet" \
  "https://datalab.dive.edito.eu/api/my-lab/app?path=scheduled-glonet-forecast-data-orchestration"

curl --fail-with-body -X PUT \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $EDITO_ACCESS_TOKEN" \
  -H "ONYXIA-REGION: WAW3-1" \
  -H "ONYXIA-PROJECT: project-glonet" \
  "https://datalab.dive.edito.eu/api/my-lab/app" \
  --data-binary @- << EOF
{
  "catalogId": "process-playground",
  "packageName": "glonet-forecast-data-orchestration",
  "packageVersion": "0.0.23",
  "name": "scheduled-glonet-forecast-data-orchestration",
  "friendlyName": "scheduled-glonet-forecast-data-orchestration",
  "share": false,
  "options": {
    "copernicusMarine": {
      "username": "$COPERNICUSMARINE_SERVICE_USERNAME",
      "password": "$COPERNICUSMARINE_SERVICE_PASSWORD"
    },
    "editoApi": {
      "accessToken": "$EDITO_ACCESS_TOKEN"
    },
    "jobResults": {
      "endpoint_url_location": "",
      "output_location": ""
    },
    "resources": {
      "limits": {
        "cpu": "7200m",
        "memory": "32Gi",
        "nvidia.com/gpu": "1"
      },
      "requests": {
        "cpu": "7200m",
        "memory": "32Gi"
      }
    },
    "s3": {
      "accessKeyId": "${AWS_ACCESS_KEY_ID}",
      "defaultRegion": "waw3-1",
      "enabled": true,
      "endpoint": "$AWS_S3_ENDPOINT",
      "secretAccessKey": "$AWS_SECRET_ACCESS_KEY",
      "sessionToken": "$AWS_SESSION_TOKEN"
    },
    "startupProbe": {},
    "catalogType": "Process"
  },
  "dryRun": false
}
EOF
