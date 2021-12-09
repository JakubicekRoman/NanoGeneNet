# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:18:58 2021

@author: jakubicek
"""



import matplotlib.pyplot as plt
import numpy as np


s = sample.squeeze().detach().cpu().numpy()[1500:7000] 

plt.figure
plt.plot( s )
# plt.plot( test_ACC )
plt.ylim([10, 100])
plt.show()   

plt.figure
plt.plot( s )
plt.plot( np.arange(4000,4500), s[4000:4500])
plt.ylim([10, 100])
plt.show()  

plt.figure
plt.plot( s )
plt.plot( np.arange(4000,4500), s[4000:4500] , color='red')
plt.ylim([10, 100])
plt.show()  


s = sample.squeeze().detach().cpu().numpy()[6900:11000] 

plt.figure
plt.plot( s )
# plt.plot( test_ACC )
plt.ylim([10, 100])
plt.show()   

plt.figure
plt.plot( s )
plt.plot( np.arange(200,1000), s[200:1000])
plt.ylim([10, 100])
plt.show()  

plt.figure
plt.plot( s )
plt.plot( np.arange(200,1000), s[200:1000] ,  color='green')
plt.ylim([10, 100])
plt.show()  



s = sample.squeeze().detach().cpu().numpy()[20000:29000] 

plt.figure
plt.plot( s )
# plt.plot( test_ACC )
plt.ylim([10, 100])
plt.show()   

plt.figure
plt.plot( s )
plt.plot( np.arange(2800,3300), s[2800:3300] )
plt.ylim([10, 100])
plt.show()  

plt.figure
plt.plot( s )
plt.plot( np.arange(2800,3300), s[2800:3300] , color='yellow')
plt.ylim([10, 100])
plt.show()  