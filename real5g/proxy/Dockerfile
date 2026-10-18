FROM golang:1.23-alpine AS build
WORKDIR /src
COPY go.mod main.go ./
RUN go mod tidy
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/sbi-fault-proxy ./main.go

FROM alpine:3.20
COPY --from=build /out/sbi-fault-proxy /usr/local/bin/sbi-fault-proxy
ENTRYPOINT ["/usr/local/bin/sbi-fault-proxy"]
