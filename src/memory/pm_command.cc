#include "memory/shm_file.h"

#include "base/init.h"
#include "base/string.h"

DEFINE_string(command, "----", "reinit");
DEFINE_int32(numa_id, 0, "");

int main(int argc, char** argv) {
  base::Init(&argc, &argv);
  if (FLAGS_command == "reinit") {
    base::PMMmapRegisterCenter::GetConfig().backend =
        base::PMMmapRegisterCenter::Backend::kDevDax;
    base::PMMmapRegisterCenter::GetConfig().numa_id = FLAGS_numa_id;
    base::PMMmapRegisterCenter::GetInstance()->ReInitialize();
    auto command = base::SFormat("rm -rf /media/aep0/*;"
                                 "rm -rf /media/aep1/*;");
    LOG(WARNING) << command;
    system(command.c_str());
  } else {
    LOG(FATAL) << "false command";
  }
  return 0;
}
