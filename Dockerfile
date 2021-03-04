#Build stage.
FROM ubuntu:20.04

WORKDIR /build

RUN apt-get -y update

RUN apt-get -y install git

RUN npm install opencv-build opencv4nodejs

RUN npm ci

COPY package.json package-lock.json ./

COPY . .

#Runtime container
FROM node:15.8.0-alpine3.10

USER node

RUN mkdir /home/node/jedi

WORKDIR /home/node/jedi

COPY --from=0 --chown=node:node /build .

CMD ["node", "server.js"]