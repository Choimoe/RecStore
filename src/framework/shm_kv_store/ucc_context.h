#pragma once

#include <ucc/api/ucc.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace recstore {
namespace framework {

class UCCContext {
public:
  static UCCContext& GetInstance();

  void Barrier();

  std::vector<char> Allgatherv(const std::vector<char>& send_buffer);

  int GetRank() const { return rank_; }

  int GetWorldSize() const { return world_size_; }

  UCCContext(const UCCContext&)            = delete;
  UCCContext& operator=(const UCCContext&) = delete;

private:
  UCCContext();
  ~UCCContext();

  void check_ucc_status(ucc_status_t status, const std::string& message);

  ucc_lib_h lib_         = nullptr;
  ucc_context_h context_ = nullptr;
  ucc_team_h team_       = nullptr;

  int rank_       = 0;
  int world_size_ = 1;
};

} // namespace framework
} // namespace recstore