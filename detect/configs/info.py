def log_trian_info(cfgdic):

    Parameters  = "Parameters "
    Value       = "Value"
    print("+"+20 * "—" +"+"+20* "—"+"+")
    print("|"f"{Parameters:^20}|{Value:^20}|")
    print("+"+20 * "—" +"+"+20* "—"+"+")
    for k,v in cfgdic.items():        
        print("|"f"{k:^20}|{v:^20}|")
    print("+"+20 * "—" +"+"+20* "—"+"+")