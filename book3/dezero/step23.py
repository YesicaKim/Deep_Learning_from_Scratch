# Add import path for the dezero directory.
if '__file__' in globals():   # 터미널에서 python 명령으로 실행하면 __file__변수가 정의되어 있음. 
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 현재 파일의 부모 디렉토리를 모듈 검색 경로에 추가함

import numpy as np
# from dezero.core_simple import Variable
from dezero import Variable


x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)