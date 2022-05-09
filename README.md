# Diplomatic_project

This is my Diplomatic Project repo.

It contains the the code needed to build and lauch the app shown in my channel
https://www.youtube.com/channel/UCdaFWlaoeB5f-BgTzcOZekw

To run this you need to build the base_image docker Image by running
"docker build --rm -f "base_image\Dockerfile" -t base_image:latest "base_image" "

or 

docker build --rm -f "base_image/Dockerfile" -t base_image:latest "base_image"

on linux.

Then after its built and saved (about 7.14 GB image)

You need to run the docker compose file by running 
"docker-compose -f "docker-compose.yml" up --build"

And you shall have them running and communicating.

You can also run each one individually but it is not recommended since they need eachother to serve a prupose.

For any details please contact me at panikoschristou@yahoo.com

Copyright 2022 Panikos Christou

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
