from flask import Flask 
from flask_restful import Resource, Api, reqparse 
import minimum
import numpy as np 
import json 


app = Flask("PowerPlantAPI")
api = Api(app) 

parser  = reqparse.RequestParser() 
parser.add_argument("load", type = int)
parser.add_argument("fuels", type = dict, action = "append") 
parser.add_argument("powerplants", type = dict, action = "append" )

gas = "gas(euro/MWh)"
ker = "kerosine(euro/MWh)"
co2 = "co2(euro/ton)"
wind = "wind(%)"

PPtype2Fuel = {
    "gasfired" : gas, 
    "turbojet" : ker, 
    "windturbine" : wind
}
class Server(Resource) : 
    def post(self): 
        args = parser.parse_args()
        load = args["load"] 
        fuels = args["fuels"][0]
        powerplants = args["powerplants"]
        
        power_limits = np.array([( (pp["pmin"], pp["pmax"]) if pp["type"] != "windturbine" else (pp["pmax"] * (fuels[wind]/100), pp["pmax"] * (fuels[wind]/100)) ) for pp in powerplants]) 
        cost_factor = np.array([ ( (fuels[PPtype2Fuel[pp["type"]]]/ pp["efficiency"]) + ( (fuels[co2] * 0.3) if pp["type"] == "gasfired" else 0 ) ) if pp["type"] != "windturbine" else 0  for pp in powerplants]) 
        best_policy = minimum.find_global_minimum(power_limits, cost_factor ,load)

        return [{"name" : powerplants[i]["name"], "p" : best_policy[i]} for i in range(len(powerplants))]



api.add_resource(Server, "/productionplan")

if __name__ == "__main__" : 
    app.run(host = "0.0.0.0", port = 8888)