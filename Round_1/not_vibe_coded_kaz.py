from datamodel import OrderDepth, TradingState, Order, Trade
from typing import Dict, List, Tuple, Any
import json
import math
import numpy as np


#(root_holding, root_holding_deviation        )
initialization_data = []

### Trader Class, our main class ###
class Trader:

    #Initializing harcoded values, inputs are the tunable parameters
    def __init__(self, initialization_data) -> None:
        self.hardcoded : Dict[str : Dict[str :]] =  {
            # Tuned values for each good
            "INTARIAN_PEPPER_ROOT": {
                "order_limit" : 80,
                "fair_value" : None, #this needs to be initialized on the first tick
                "trend" : 0.1, #Perchance not hardcode this
                "holding" : initialization_data[0],
                "holding_deviation" : initialization_data[1],
                
                },
            "ASH_COATED_OSMIUM": {},
        }
        
    ### HELPER FUNCTIONS ###
        
        

    ### TRADERS ###
    def root_trader(self, time : int, past_data : str, position : int, orders : Order) -> Order:
        #Walls
        try: bid_wall = min([x for x,_ in orders.buy_orders.items()])
        except: pass
        
        try: ask_wall = max([x for x,_ in orders.sell_orders.items()])
        except: pass

        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass
    
        #Spread
        spread = ask_wall - bid_wall
        
        #Current Regime : 0 -> buy / 1 -> market making / 2 -> sell
        if abs(self.hardcoded["INTERIAN_PEPPER_ROOTS"]):
            pass
        
        #Pass orders
        
        
        return None #orders
    
    def osmium_trader(self, time : int, past_data : str, position : int, orders : Order) -> Order:
        #Spread
        #Asks
        #Bids
        #Current Regime : 0 -> buy / 1 -> market making / 2 -> sell
        
        #Pass orders
        
        
        return None #orders



    ### MAIN FINAL FUNCTION ###
    def run(self, state: TradingState) -> Dict["str" : Order]:
        #Extract data from TradingState
        past_data : str = state.traderData
        time : int = state.timestamp
        positions : Dict[str : int] = state.positions
        orders : Dict[str : OrderDepth]= state.order_depth
        
        #Construct our orders
        our_orders : Dict["str" : Order] = {}
        our_orders["INTARIAN_PEPPER_ROOTS"] = self.root_trader(time, past_data, positions["INTARIAN_PEPPER_ROOTS"], orders["INTARIAN_PEPPER_ROOTS"])
        #our_orders["ASH_COATED_OSMIUM"] = self.osmium_trader(time, past_data, positions["ASH_COATED_OSMIUM"], orders["ASH_COATED_OSMIUM"])
        
        return our_orders
