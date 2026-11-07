function(asr_patch_grpc_ssl)
# gRPC v1.83.0 predates OpenSSL 4 const getters and opaque ASN.1 strings.
# Use getters rather than structure fields; auto preserves the getter's
# constness on both OpenSSL 3 and 4. No compiler suppression or const_cast.
set(_grpc_ssl "${grpc_SOURCE_DIR}/src/core/tsi/ssl_transport_security.cc")
file(SHA256 "${_grpc_ssl}" _grpc_ssl_hash)
if(_grpc_ssl_hash STREQUAL "b97c3f5bfbdae0da1d944ad963f67fbab8778207c0d1f80391c2ac2c6decb836")
  return()
endif()
if(NOT _grpc_ssl_hash STREQUAL "1d6d37e08cbea6961858b6c41c63e7674756bfc863363fbdaacb8c50b8048ce6")
  message(FATAL_ERROR "gRPC OpenSSL compatibility patch needs review for changed source")
endif()
file(READ "${_grpc_ssl}" _grpc_ssl_text)
string(REPLACE [=[  X509_NAME_ENTRY* common_name_entry = nullptr;
]=] [=[]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(REPLACE [=[  ASN1_STRING* common_name_asn1 = nullptr;
]=] [=[]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(REPLACE [=[X509_NAME* subject_name = X509_get_subject_name(cert);]=] [=[auto* subject_name = X509_get_subject_name(cert);]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(REPLACE [=[  common_name_entry = X509_NAME_get_entry(]=] [=[  auto* common_name_entry = X509_NAME_get_entry(]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(REPLACE [=[  common_name_asn1 = X509_NAME_ENTRY_get_data(]=] [=[  auto* common_name_asn1 = X509_NAME_ENTRY_get_data(]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(REPLACE [=[subject_alt_name->d.iPAddress->length]=] [=[ASN1_STRING_length(subject_alt_name->d.iPAddress)]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(REPLACE [=[subject_alt_name->d.iPAddress->data]=] [=[ASN1_STRING_get0_data(subject_alt_name->d.iPAddress)]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(REPLACE [=[      root_name = X509_get_subject_name(root);
      if (root_name == nullptr)]=] [=[      auto* subject_name = X509_get_subject_name(root);
      if (subject_name == nullptr)]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(REPLACE [=[root_name = X509_NAME_dup(root_name);]=] [=[root_name = X509_NAME_dup(subject_name);]=] _grpc_ssl_text "${_grpc_ssl_text}")
string(SHA256 _grpc_ssl_patched_hash "${_grpc_ssl_text}")
if(NOT _grpc_ssl_patched_hash STREQUAL "b97c3f5bfbdae0da1d944ad963f67fbab8778207c0d1f80391c2ac2c6decb836")
  message(FATAL_ERROR "gRPC OpenSSL compatibility patch produced unexpected source")
endif()
file(WRITE "${_grpc_ssl}" "${_grpc_ssl_text}")
endfunction()
asr_patch_grpc_ssl()

# The CRL issuer getter has the same const change as certificate subjects.
set(_grpc_crl "${grpc_SOURCE_DIR}/src/core/credentials/transport/tls/grpc_tls_crl_provider.cc")
file(SHA256 "${_grpc_crl}" _grpc_crl_hash)
if(_grpc_crl_hash STREQUAL "0ba70a810117902d46bfbea2dfaaace7af4db265b6ed80a1982f76171a61eeab")
  file(READ "${_grpc_crl}" _grpc_crl_text)
  string(REPLACE "X509_NAME* issuer = X509_CRL_get_issuer(crl);"
                 "auto* issuer = X509_CRL_get_issuer(crl);"
                 _grpc_crl_text "${_grpc_crl_text}")
  file(WRITE "${_grpc_crl}" "${_grpc_crl_text}")
elseif(NOT _grpc_crl_hash STREQUAL "c24a7ff65a0a091dfcc859a44bba6119c96e041c84e4b725860535e65e26f8d1")
  message(FATAL_ERROR "gRPC OpenSSL CRL patch needs review for changed source")
endif()

# Certificate/CRL comparison and authority key identifiers only read getters.
set(_grpc_utils "${grpc_SOURCE_DIR}/src/core/tsi/ssl_transport_security_utils.cc")
file(SHA256 "${_grpc_utils}" _grpc_utils_hash)
if(_grpc_utils_hash STREQUAL "fc4fabd7fc1032e3e77c12c9ed93435996723130be9343fc88052dba66c0593d")
  file(READ "${_grpc_utils}" _grpc_utils_text)
  string(REPLACE "X509_NAME* cert_issuer_name =" "auto* cert_issuer_name =" _grpc_utils_text "${_grpc_utils_text}")
  string(REPLACE "X509_NAME* crl_issuer_name =" "auto* crl_issuer_name =" _grpc_utils_text "${_grpc_utils_text}")
  string(REPLACE "X509_NAME* issuer =" "auto* issuer =" _grpc_utils_text "${_grpc_utils_text}")
  string(REPLACE "ASN1_OCTET_STRING* akid = nullptr;" "const ASN1_OCTET_STRING* akid = nullptr;" _grpc_utils_text "${_grpc_utils_text}")
  file(WRITE "${_grpc_utils}" "${_grpc_utils_text}")
elseif(NOT _grpc_utils_hash STREQUAL "c172aab029fda86099e9a68d7745774a0bbba7eb357c6a333c6f1b691fda2eb9")
  message(FATAL_ERROR "gRPC OpenSSL utilities patch needs review for changed source")
endif()
