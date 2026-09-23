FROM node:18-alpine as build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
# 修复兼容性: 新版 nginx:alpine（Alpine apk-tools 3.x）的 10-listen-on-ipv6 脚本
# 会因 `apk manifest` 命令挂起，导致 nginx 永远无法启动（容器卡在 Up 但 80 端口无监听）。
# 本项目 nginx.conf 已显式 listen 80，无需该脚本的 IPv6 自动配置，直接移除。
RUN rm -f /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
