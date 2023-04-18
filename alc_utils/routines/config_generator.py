# this code includes functions to cross product of all the
# configuration parameters for campaigns.


def generateConfigs(exptParams, campaignParams):
    configs = {}
    eparams = exptParams
    cparams = campaignParams

    [eparamInitial, cparamlists, cparamsindex,
        numconfig] = getCampaignParamLists(eparams, cparams)

    for i in range(0, numconfig):
        cparamVals = getCampaignParamValues(
            cparamlists, cparamsindex, i, numconfig)
        [configname, configvalue] = generateConfig(
            i, eparamInitial, cparamVals)
        configs[configname] = configvalue
    return configs


def getCampaignParamLists(eparams, cparams):
    cparamskeys = cparams.keys()
    eparamsOutput = eparams.copy()
    cparamslist = {}
    cparamsindex = {}
    num = 1
    for k in cparamskeys:
        if (type(cparams[k]) is not list) or (len(cparams[k]) == 1):
            y = {k: cparams[k]}
            eparamsOutput.update(y)
        else:
            cparamslist[k] = cparams[k]
            cparamsindex[k] = (num, len(cparamslist[k]))
            num = num * len(cparamslist[k])

    return [eparamsOutput, cparamslist, cparamsindex, num]


def getCampaignParamValues(cparamlists, cparamsindex, idx, numconfig):
    cparamskeys = cparamlists.keys()
    cparams = {}
    for k in cparamskeys:
        (s, n) = cparamsindex[k]
        values = cparamlists[k]
        a = int(idx / s)
        b = a % n
        cparams[k] = values[b]
    return cparams


def generateConfig(idx, eparamInitial, cparamVals):
    configParams = eparamInitial.copy()
    configParams.update(cparamVals)
    configname = str(idx)
    return [configname, configParams]


def test_run():
    exptParams = '''{"param1": 1, "param2":"abc", "param3":true, "param4": 0,  "param5":"hello1"}'''
    campParams = '''{"param4":[0,1,2,3], "param5":["hello1", "hello2"], "param6": "xyz", "param7":1}'''
    configVals = generateConfigs(exptParams, campParams)


def run(exptParams, campaignParams):
    print 'Expt Params :' + exptParams
    print 'CampaignParams:' + campaignParams
    configVals = generateConfigs(exptParams, campaignParams)
