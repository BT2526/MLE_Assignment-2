"""Helper: compute application date (label date - 6 months) from command line arg."""
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
label_date = datetime.strptime(sys.argv[1], "%Y-%m-%d")
app_date = label_date - relativedelta(months=6)
print(app_date.strftime("%Y-%m-%d"))
