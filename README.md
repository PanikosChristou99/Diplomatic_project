# Diplomatic_project

This is my Diplomatic Project repo.

It contains the the code needed to build and lauch the app shown in my channel
https://www.youtube.com/channel/UCdaFWlaoeB5f-BgTzcOZekw

To run this you need to build the base_image docker Image by running
"docker build --rm -f "base_image\Dockerfile" -t base_image:latest "base_image" "

Then after its built and saved (about 7.14 GB image)

You need to run the docker compose file by running 
"docker-compose -f "docker-compose.yml" up -d --build"

And you shall have them running and communicating.

You can also run each one individually but it is not recommended since they need eachother to serve a prupose.

For any details please contact me at panikoschristou@yahoo.com
