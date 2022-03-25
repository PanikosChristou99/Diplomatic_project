# !/bin/bash
docker-compose -f "./docker-compose.yml" up --build -d &
echo "Sleeping for 1800 secs"
sleep 1800s
docker-compose stop
