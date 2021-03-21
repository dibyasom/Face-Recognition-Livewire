#Build stage.
FROM node:15.11.0-stretch

WORKDIR /build

# RUN npm install opencv-build opencv4nodejs

COPY package.json package-lock.json ./

RUN npm ci

COPY . .

#Runtime container
FROM node:15.8.0-alpine3.10

USER node

RUN mkdir /home/node/jedi

WORKDIR /home/node/jedi

COPY --from=0 --chown=node:node /build .

CMD ["node", "server.js"]