// Explicitly instantiate the std::string-returning member functions of
// boost::cpp_regex_traits_implementation<char> with the CXX11 ABI
// (_GLIBCXX_USE_CXX11_ABI=1).
//
// folly is built with the new C++ ABI and therefore references the
// [abi:cxx11]-mangled transform()/transform_primary() symbols. The system
// boost_regex shared library, however, was built with the old libstdc++ ABI
// and only exports the non-cxx11 instantiations. Because std::string is
// ABI-tagged the two manglings are distinct symbols, so the cxx11 ones would
// otherwise stay undefined and break dlopen of _recstore_ops.so. This
// translation unit supplies them; it is compiled as a regular source of
// recstore_torch_ops, which already uses _GLIBCXX_USE_CXX11_ABI=1.
//
// Version-agnostic: BOOST_REGEX_DETAIL_NS (defined by boost >= 1.56) expands to
// the correct re_detail_XXXXXX namespace. Older boost (e.g. 1.53) lacks the
// macro and uses the unversioned re_detail namespace, so fall back to that.
#include <locale>
#include <boost/regex.hpp>

#ifndef BOOST_REGEX_DETAIL_NS
#  define BOOST_REGEX_DETAIL_NS re_detail
#endif

template boost::BOOST_REGEX_DETAIL_NS::cpp_regex_traits_implementation<
    char>::string_type
boost::BOOST_REGEX_DETAIL_NS::cpp_regex_traits_implementation<char>::transform(
    const char*, const char*) const;

template boost::BOOST_REGEX_DETAIL_NS::cpp_regex_traits_implementation<
    char>::string_type
boost::BOOST_REGEX_DETAIL_NS::cpp_regex_traits_implementation<
    char>::transform_primary(const char*, const char*) const;
