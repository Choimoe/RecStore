#ifndef CCEH_FILE_H_
#define CCEH_FILE_H_

#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#define PAGE_SIZE 4096
typedef uint64_t PageID_t;
const PageID_t INVALID_PAGE = -1;

class FileManager {
public:
  FileManager(const std::string &file_path) {
    bool exists = (access(file_path.c_str(), F_OK) != -1);
    fd =
        open(file_path.c_str(), O_RDWR | O_CREAT | O_DIRECT, S_IRUSR | S_IWUSR);
    if (fd < 0) {
      throw std::runtime_error("Failed to open file: " + file_path);
    }

    if (exists) {
      struct stat file_stat;
      fstat(fd, &file_stat);
      next_page_id = file_stat.st_size / PAGE_SIZE;
    } else {
      next_page_id = 0;
    }
  }

  ~FileManager() {
    if (fd >= 0) {
      close(fd);
    }
  }

  PageID_t AllocatePage() {
    // Simple sequential allocation
    PageID_t new_page_id = next_page_id++;
    // Extend file size
    if (pwrite(fd, &empty_page, PAGE_SIZE, new_page_id * PAGE_SIZE) !=
        PAGE_SIZE) {
      throw std::runtime_error("Failed to extend file for new page");
    }
    return new_page_id;
  }

  void ReadPage(PageID_t page_id, char *buffer) {
    if (pread(fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE) != PAGE_SIZE) {
      throw std::runtime_error("Failed to read page " +
                               std::to_string(page_id));
    }
  }

  void WritePage(PageID_t page_id, const char *buffer) {
    if (pwrite(fd, buffer, PAGE_SIZE, page_id * PAGE_SIZE) != PAGE_SIZE) {
      throw std::runtime_error("Failed to write page " +
                               std::to_string(page_id));
    }
  }

  // A simplified buffer manager for getting and pinning pages
  // In a real system, this would be a proper buffer pool manager
  template <typename T> T *GetPage(PageID_t page_id) {
    void *buffer;
    if (posix_memalign(&buffer, PAGE_SIZE, PAGE_SIZE) != 0) {
      throw std::runtime_error("Failed to allocate aligned memory for page");
    }
    ReadPage(page_id, reinterpret_cast<char *>(buffer));
    return reinterpret_cast<T *>(buffer);
  }

  // Unpin a page, if dirty, write it back
  void Unpin(PageID_t page_id, const void *page_data, bool is_dirty) {
    if (is_dirty) {
      WritePage(page_id, reinterpret_cast<const char *>(page_data));
    }
    free(const_cast<void *>(page_data));
  }

  int fd;

private:
  PageID_t next_page_id;
  char empty_page[PAGE_SIZE] = {0};
};

#endif // CCEH_FILE_H_