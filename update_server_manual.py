import os,json

APPS = ['swaptions','ferret','svm','nn','facedetect','bodytrack']
#APPS=['ferret']

def getServerLoc(app):
    return '/var/www/html/rapid_server/storage/apps/algaesim-'+app+'/'

def getLocalLoc(app):
    return '/home/liuliu/Research/rapid_m_backend_server/outputs/'+app+'/'

def rewriteLoc(app):
    with open(getLocalLoc(app)+app+'.json') as f:
        f = json.load(f)
        mp = f['model_params']
        for b_name, info in mp.items():
            info['file']=getServerLoc(app)+b_name
        dest_file = open(getServerLoc(app)+'profile.json','w')
        json.dump(f,dest_file,indent=2)

def copyModels(app):
    files = getLocalLoc(app)+app+'*'
    dest_loc = getServerLoc(app)
    # clear all old models
    os.system('rm '+dest_loc+app+'*')
    # copy all new models
    cmd = 'cp '+files+" "+dest_loc
    os.system(cmd)
    # clear the redundant json
    os.system('rm '+dest_loc+app+'.json')

for app in APPS:
    # move models to server
    copyModels(app)
    # update the profile.json
    rewriteLoc(app)
# update the cost/mv
    os.system('./update_profile.sh')
