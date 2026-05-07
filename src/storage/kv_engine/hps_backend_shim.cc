#include <hps/hier_parameter_server_base.hpp>

#include <sstream>

namespace HugeCTR {

std::string HierParameterServerBase::make_tag_name(const std::string& model_name,
                                                   const std::string& embedding_table_name,
                                                   const bool /*check_arguments*/) {
  std::ostringstream os;
  os << PS_EMBEDDING_TABLE_TAG_PREFIX << '.';
  os << model_name << '.';
  os << embedding_table_name;
  return os.str();
}

HierParameterServerBase::~HierParameterServerBase() = default;

}  // namespace HugeCTR
