// Exercise the vendored TLS code against the actual linked OpenSSL, including
// the certificate getters and opaque ASN.1 IP strings changed in OpenSSL 4.
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <grpc/grpc.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include "src/core/tsi/ssl_transport_security.h"
#include "src/core/credentials/transport/tls/grpc_tls_crl_provider.h"
#include "src/core/tsi/transport_security.h"
#include "src/core/tsi/ssl_transport_security_utils.h"

static void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

int main() {
  grpc_init();
  EVP_PKEY* key = EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "prime256v1");
  require(key != nullptr, "generate fixture key");
  X509* cert = X509_new();
  require(cert != nullptr, "allocate certificate");
  require(X509_set_version(cert, 2) == 1, "set version");
  require(ASN1_INTEGER_set(X509_get_serialNumber(cert), 1) == 1, "set serial");
  require(X509_gmtime_adj(X509_getm_notBefore(cert), -60) != nullptr, "set start");
  require(X509_gmtime_adj(X509_getm_notAfter(cert), 3600) != nullptr, "set expiry");
  require(X509_set_pubkey(cert, key) == 1, "set public key");
  X509_NAME* name = X509_NAME_new();
  require(name != nullptr, "allocate subject");
  require(X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
      reinterpret_cast<const unsigned char*>("asr.test"), -1, -1, 0) == 1, "set CN");
  require(X509_set_subject_name(cert, name) == 1, "set subject");
  require(X509_set_issuer_name(cert, name) == 1, "set issuer");
  {
    X509_CRL* crl = X509_CRL_new();
    require(crl != nullptr && X509_CRL_set_issuer_name(crl, name) == 1, "set CRL issuer");
    unsigned char* der = nullptr;
    const int der_length = i2d_X509_NAME(name, &der);
    require(der_length > 0, "encode expected CRL issuer");
    const std::string expected(reinterpret_cast<char*>(der), static_cast<size_t>(der_length));
    OPENSSL_free(der);
    auto parsed = grpc_core::experimental::CrlImpl::Create(crl);
    require(parsed.ok(), "extract CRL issuer");
    require((*parsed)->Issuer() == expected, "preserve CRL issuer bytes");
    auto issuer = tsi::IssuerFromCert(cert);
    require(issuer.ok() && *issuer == expected, "preserve certificate issuer bytes");
    require(tsi::VerifyCrlCertIssuerNamesMatch(crl, cert), "match CRL issuer to certificate");
    require(!tsi::AkidFromCertificate(cert).ok(), "reject missing certificate AKID");
    require(!tsi::AkidFromCrl(crl).ok(), "reject missing CRL AKID");
    // DER AuthorityKeyIdentifier containing a one-byte keyIdentifier.
    const unsigned char akid_der[] = {0x30, 0x03, 0x80, 0x01, 0x01};
    ASN1_OCTET_STRING* octets = ASN1_OCTET_STRING_new();
    require(octets && ASN1_OCTET_STRING_set(octets, akid_der, sizeof(akid_der)) == 1,
            "create authority key identifier");
    der = nullptr;
    const int akid_length = i2d_ASN1_OCTET_STRING(octets, &der);
    require(akid_length > 0, "encode expected AKID");
    const std::string expected_akid(reinterpret_cast<char*>(der), static_cast<size_t>(akid_length));
    OPENSSL_free(der);
    X509_EXTENSION* extension = X509_EXTENSION_create_by_NID(
        nullptr, NID_authority_key_identifier, 0, octets);
    require(extension && X509_add_ext(cert, extension, -1) == 1 &&
            X509_CRL_add_ext(crl, extension, -1) == 1, "set authority key identifiers");
    X509_EXTENSION_free(extension);
    ASN1_OCTET_STRING_free(octets);
    auto cert_akid = tsi::AkidFromCertificate(cert);
    auto crl_akid = tsi::AkidFromCrl(crl);
    require(cert_akid.ok() && *cert_akid == expected_akid, "preserve certificate AKID");
    require(crl_akid.ok() && *crl_akid == expected_akid, "preserve CRL AKID");
  }
  X509_NAME_free(name);
  X509_EXTENSION* san = X509V3_EXT_conf_nid(nullptr, nullptr, NID_subject_alt_name,
                                         "DNS:asr.test,IP:127.0.0.1,IP:::1");
  require(san != nullptr && X509_add_ext(cert, san, -1) == 1, "set SANs");
  X509_EXTENSION_free(san);
  require(X509_sign(cert, key, EVP_sha256()) > 0, "sign certificate");
  BIO* bio = BIO_new(BIO_s_mem());
  require(bio != nullptr && PEM_write_bio_X509(bio, cert) == 1, "serialize certificate");
  char* data = nullptr;
  const long length = BIO_get_mem_data(bio, &data);
  require(length > 0, "read certificate");
  std::string pem(data, static_cast<size_t>(length));
  BIO_free(bio);
  X509_free(cert);
  bio = BIO_new(BIO_s_mem());
  require(bio != nullptr && PEM_write_bio_PrivateKey(bio, key, nullptr, nullptr, 0,
                                                    nullptr, nullptr) == 1, "serialize key");
  const long key_length = BIO_get_mem_data(bio, &data);
  require(key_length > 0, "read key");
  std::string key_pem(data, static_cast<size_t>(key_length));
  BIO_free(bio);
  EVP_PKEY_free(key);

  tsi_peer peer{};
  require(tsi_ssl_extract_x509_subject_names_from_pem_cert(pem.c_str(), &peer) == TSI_OK,
          "extract certificate identities");
  require(tsi_ssl_peer_matches_name(&peer, "asr.test"), "DNS SAN match");
  require(tsi_ssl_peer_matches_name(&peer, "127.0.0.1"), "IPv4 SAN match");
  require(tsi_ssl_peer_matches_name(&peer, "::1"), "IPv6 SAN match");
  require(!tsi_ssl_peer_matches_name(&peer, "127.0.0.2"), "reject incorrect IP");
  require(!tsi_ssl_peer_matches_name(&peer, "other.test"), "reject incorrect DNS name");
  const auto* cn = tsi_peer_get_property_by_name(&peer, TSI_X509_SUBJECT_COMMON_NAME_PEER_PROPERTY);
  require(cn && std::string(cn->value.data, cn->value.length) == "asr.test", "preserve CN");
  tsi_peer_destruct(&peer);
  auto* roots = tsi_ssl_root_certs_store_create(pem.c_str());
  require(roots != nullptr, "load root certificate");
  tsi_ssl_root_certs_store_destroy(roots);
  // Client authentication also copies root subject names into the CA list.
  tsi_ssl_server_handshaker_factory* factory = nullptr;
  std::vector<tsi_ssl_pem_key_cert_pair> pairs;
  pairs.emplace_back(key_pem, pem);
  require(tsi_create_ssl_server_handshaker_factory(pairs, pem.c_str(), 1,
              nullptr, nullptr, 0, &factory) == TSI_OK, "load client CA names");
  tsi_ssl_server_handshaker_factory_unref(factory);
  require(tsi_ssl_extract_x509_subject_names_from_pem_cert("not a certificate", &peer)
              == TSI_INVALID_ARGUMENT, "reject malformed certificate");
  grpc_shutdown();
  std::cout << "OpenSSL certificate identity and root loading passed\n";
}
