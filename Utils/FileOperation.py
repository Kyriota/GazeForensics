import os
from subprocess import check_output

def fileWalk(directory):
    dirList = []
    for root, _, files in os.walk(directory, topdown=False):
        for name in files:
            dirList.append(os.path.join(root[len(directory):], name))
    # get rid of '/' in the beginning
    for i in range(len(dirList)):
        if dirList[i][0] == '/':
            dirList[i] = dirList[i][1:]
    return dirList

def ls(directory=None):
    if directory:
        cmdOut = check_output('cd ' + directory + ' && ls', shell=True)
    else:
        cmdOut = check_output('ls', shell=True)
    return cmdOut.decode().split('\n')[:-1]

def ExecTryCmd(cmd):
    try:
        check_output(cmd, shell=True)
    except:
        pass

def fileExist(path):
    return os.path.exists(path)

def mkdir(directory):
    if not fileExist(directory):
        ExecTryCmd('mkdir ' + directory)

def rm(directory, r=False):
    if fileExist(directory):
        ExecTryCmd(('rm -r ' if r else 'rm ') + directory)

def mv(src, dst):
    ExecTryCmd('mv ' + src + ' ' + dst)

def cp(src, dst, r=False):
    ExecTryCmd(('cp -r ' if r else 'cp ') + src + ' ' + dst)

def uncompress_tar(tar_path, dst_path=None, parameters='xf'):
    if dst_path:
        ExecTryCmd('tar -' + parameters + ' ' + tar_path + ' -C ' + dst_path)
    else:
        ExecTryCmd('tar -' + parameters + ' ' + tar_path)