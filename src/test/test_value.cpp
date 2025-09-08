#include "../storage/hybrid/value.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>
#include <unistd.h>  
#include <fcntl.h>   

namespace {

constexpr size_t kSmallSize = 1024;  // 1KB
constexpr size_t kLargeSize = 2 * 1024 * 1024;  // 2MB

class ValueManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建临时文件路径
        // char shm_tmpl[] = "/tmp/shm_XXXXXX";
        char ssd_tmpl[] = "/tmp/ssd_XXXXXX";
        
        // int shm_fd = mkstemp(shm_tmpl);
        // ASSERT_NE(shm_fd, -1) << "Failed to create SHM temp file";
        // close(shm_fd);
        
        int ssd_fd = mkstemp(ssd_tmpl);
        ASSERT_NE(ssd_fd, -1) << "Failed to create SSD temp file";
        close(ssd_fd);
        // shm_path_ = shm_tmpl;
        ssd_path_ = ssd_tmpl;
            
        manager_ = std::make_unique<ValueManager>(
            // shm_path_, kSmallSize,  // 小容量SHM（易满）
            ssd_path_, kLargeSize   // 大容量SSD
        );
    }

    void TearDown() override {
        manager_.reset();  // 确保先销毁manager以关闭文件
        
        // 删除临时文件
        // std::remove(shm_path_.c_str());
        std::remove(ssd_path_.c_str());
    }

    // std::string shm_path_;
    std::string ssd_path_;
    std::unique_ptr<ValueManager> manager_;
};

// 测试写入/读取小数据（应在SHM中）
TEST_F(ValueManagerTest, WriteAndReadSmallValue) {
    const std::string test_data = "Hello, SHM!";
    
    // 写入数据
    auto ptr = manager_->WriteValue(test_data);
    ASSERT_TRUE(ptr) << "Write failed";

    // 读取验证
    std::string result = manager_->RetrieveValue(ptr);
    ASSERT_EQ(result, test_data);
}

// 测试写入大容量数据（应溢出到SSD）
TEST_F(ValueManagerTest, WriteLargeValueToSSD) {
    // 生成大测试数据 (1.5MB > SHM容量)
    std::string large_data(kSmallSize + 1024, 'X');
    
    // 写入数据
    auto ptr = manager_->WriteValue(large_data);
    ASSERT_TRUE(ptr) << "Write to SSD failed";
    
    // 读取验证
    std::string result = manager_->RetrieveValue(ptr);
    ASSERT_EQ(result, large_data);
}

// 测试混合读写（SHM和SSD交替）
TEST_F(ValueManagerTest, MixedTierOperations) {
    // 填充SHM
    // std::string shm_data(kSmallSize / 2, 'A');
    // auto shm_ptr = manager_->WriteValue(shm_data);
    
    // 写入大文件到SSD
    std::string ssd_data(kSmallSize + 1024, 'B');
    auto ssd_ptr = manager_->WriteValue(ssd_data);
    
    // 验证SHM数据
    // ASSERT_EQ(manager_->RetrieveValue(shm_ptr), shm_data);
    
    // 验证SSD数据
    ASSERT_EQ(manager_->RetrieveValue(ssd_ptr), ssd_data);
}

// 测试空值处理
TEST_F(ValueManagerTest, HandleEmptyValue) {
    auto ptr = manager_->WriteValue("");
    ASSERT_TRUE(ptr);
    EXPECT_TRUE(manager_->RetrieveValue(ptr).empty());
}

// 测试无效指针处理
TEST_F(ValueManagerTest, HandleInvalidPointer) {
    UnifiedPointer null_ptr;
    EXPECT_THROW(manager_->RetrieveValue(null_ptr), std::runtime_error);
}

}  // namespace