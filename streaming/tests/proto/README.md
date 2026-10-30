`health.proto` is copied verbatim from gRPC v1.51.1:
https://github.com/grpc/grpc/blob/v1.51.1/src/proto/grpc/health/v1/health.proto

Its Apache-2.0 copyright/license header is retained. The container smoke test
uses this standard wire contract to query gRPC's built-in health service,
whose descriptor is not available through reflection in every C++ build.
