#!/usr/bin/python


if __name__=='__main__':
    import sys, os
    
    paths_file=sys.argv[1]
    with open(paths_file, 'r') as fd:
        lines = fd.readlines()
    export_dict = {'GAZEBO_MODEL_PATH': [],
                   'GAZEBO_RESOURCE_PATH': [],
                   'GAZEBO_MEDIA_PATH': []}
    for line in lines:
        if line.strip()=='':
            #print 'empty line'
            continue
        #print line.strip()
        env_var, path = line.strip().split('=')
        env_var = env_var.upper()
        if env_var =='GAZEBO_PLUGIN_PATH':
            #print 'skipping plugin path'
            continue
            
        export_dict[env_var].append(path.strip('"'))
    
    export_t = 'export {env_var}={new_paths}:${env_var}'
    
    for k,v in export_dict.items():
        if len(v)==0: continue
        export_str = export_t.format(env_var=k, new_paths=':'.join(v))
        print export_str.replace("./","")
        if k=='GAZEBO_MEDIA_PATH':
            export_str = export_t.format(env_var='GAZEBO_RESOURCE_PATH', new_paths=':'.join(v))
            print export_str.replace("./","")
