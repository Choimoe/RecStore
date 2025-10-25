import sys
import ctypes
import os

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libreport.so")

try:
    lib = ctypes.CDLL(LIB_PATH)
except Exception as e:
    print(f"Error: Failed to load shared library {LIB_PATH}. Please check file existence and dependencies.")
    sys.exit(1)

lib.report.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_double
]
lib.report.restype = ctypes.c_bool

def report_metric(table: str, uid: str, metric: str, value: float) -> bool:
    return lib.report(
        table.encode('utf-8'),
        uid.encode('utf-8'),
        metric.encode('utf-8'),
        value
    )

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python report_uploader.py <table> <uid> <metric> <value>")
        print("Example: python report_uploader.py user_activity user_123 login_time 2.5")
        sys.exit(1)
    
    table = sys.argv[1]
    uid = sys.argv[2]
    metric = sys.argv[3]
    
    try:
        value = float(sys.argv[4])
    except ValueError:
        print("Error: Value must be a numeric (float) type")
        sys.exit(1)
    
    success = report_metric(table, uid, metric, value)
    
    if success:
        print(f"Data reported successfully: table='{table}', uid='{uid}', metric='{metric}', value={value}")
    else:
        print(f"Data report failed: table='{table}', uid='{uid}', metric='{metric}', value={value}")