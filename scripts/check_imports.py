import sys
modules = ['scipy','xgboost','sklearn','pandas']
print('Python executable:', sys.executable)
for m in modules:
    try:
        __import__(m)
        print(f'{m}: True')
    except Exception as e:
        print(f'{m}: False ({e.__class__.__name__})')
