#!/usr/bin/python3

from ruamel import yaml

def load_yaml_file(fileName):
    with open(fileName, 'r', encoding='utf8') as infile:
        data = yaml.load(infile, Loader=yaml.RoundTripLoader)
    return data

def update_command(data, fileName, imageName):
    if imageName == "hw":
        command_caps = [ 'bash', '-c', 'ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -hwaccel_output_format vaapi -i /var/www/archive/bbb_sunflower_1080p_30fps_normal.mp4 -vf scale_vaapi=w=2560:h=1440 -c:v h264_vaapi -b:v 15M -f flv rtmp://adi-service/hls/big_buck_bunny_2560x1440 -vf scale_vaapi=w=1920:h=1080 -c:v h264_vaapi -b:v 10M -f flv rtmp://adi-service/hls/big_buck_bunny_1920x1080 -vf scale_vaapi=w=1280:h=720 -c:v h264_vaapi -b:v 8M -f flv rtmp://adi-service/hls/big_buck_bunny_1280x720 -vf scale_vaapi=w=854:h=480 -c:v h264_vaapi -b:v 6M -f flv rtmp://adi-service/hls/big_buck_bunny_854x480 -abr_pipeline' ]
    else:
        command_caps = [ 'bash', '-c', 'ffmpeg -re -stream_loop -1 -i /var/www/archive/bbb_sunflower_1080p_30fps_normal.mp4 -vf scale=2560:1440 -c:v libsvt_hevc -b:v 15M -f flv rtmp://adi-service/hls/big_buck_bunny_2560x1440 -vf scale=1920:1080 -c:v libsvt_hevc -b:v 10M -f flv rtmp://adi-service/hls/big_buck_bunny_1920x1080 -vf scale=1280:720 -c:v libx264 -b:v 8M -f flv rtmp://adi-service/hls/big_buck_bunny_1280x720 -vf scale=854:480 -c:v libx264 -b:v 6M -f flv rtmp://adi-service/hls/big_buck_bunny_854x480 -abr_pipeline' ]
    data['spec']['template']['spec']['containers'][0].update({'args' : command_caps})
    with open(fileName, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, Dumper=yaml.RoundTripDumper, default_flow_style=False, allow_unicode=True)

def update_imageName(data, fileName, imageName, isVOD):
    if imageName == "hw" or not isVOD:
        replicas_caps = 1
    else:
        replicas_caps = 2
    data['spec']['replicas'] = replicas_caps
    data['spec']['template']['spec']['containers'][0]['image'] = "ovc_transcode_" + imageName + ":latest"
    if imageName == "hw":
        limits_caps = { 'limits': {'gpu.intel.com/i915': 1} }
        data['spec']['template']['spec']['containers'][0]['resources'] = limits_caps
    with open(fileName, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, Dumper=yaml.RoundTripDumper, default_flow_style=False, allow_unicode=True)

def add_volumeMounts(data, fileName, node_name):
    if node_name == "ad-content":
        volumemounts_caps = [ {'name': 'ad-archive',
                           'mountPath': '/var/www/archive',
                           'readOnly': False}]
    elif node_name == "ad-insertion-frontend":
        volumemounts_caps = [ {'name': 'ad-dash',
                           'mountPath': '/var/www/adinsert/dash',
                           'readOnly': False},
                          {'name': 'ad-hls',
                           'mountPath': '/var/www/adinsert/hls',
                           'readOnly': False}]
    elif node_name == "content-provider":
        volumemounts_caps = [ {'name': 'html-volume',
                           'mountPath': '/var/www/html',
                           'readOnly': False},
                          {'name': 'video-archive',
                           'mountPath': '/var/www/archive',
                           'readOnly': False},
                          {'name': 'video-dash',
                           'mountPath': '/var/www/dash',
                           'readOnly': False},
                          {'name': 'video-hls',
                           'mountPath': '/var/www/hls',
                           'readOnly': False}]
    elif node_name == "content-provider-transcode":
        volumemounts_caps = [ {'name': 'video-archive',
                           'mountPath': '/var/www/archive',
                           'readOnly': False},
                          {'name': 'video-dash',
                           'mountPath': '/var/www/dash',
                           'readOnly': False},
                          {'name': 'video-hls',
                           'mountPath': '/var/www/hls',
                           'readOnly': False}]
    elif node_name == "ad-transcode":
        volumemounts_caps = [ {'name': 'ad-dash',
                           'mountPath': '/var/www/adinsert/dash',
                           'readOnly': False},
                          {'name': 'ad-hls',
                           'mountPath': '/var/www/adinsert/hls',
                           'readOnly': False},
                          {'name': 'ad-static',
                           'mountPath': '/var/www/skipped',
                           'readOnly': False}]
    elif node_name == "cdn":
        volumemounts_caps = [ {'name': 'secrets',
                           'mountPath': '/var/run/secrets',
                           'readOnly': False}]

    data['spec']['template']['spec']['containers'][0].update({'volumeMounts' : volumemounts_caps})
    with open(fileName, "w", encoding="utf-8") as outfile:
        yaml.dump(data, outfile, Dumper=yaml.RoundTripDumper, default_flow_style=False, allow_unicode=True)

def add_volumes(data, fileName, nfs_server, node_name, adi_directory):
    if nfs_server == "localhost":
        if node_name == "ad-content":
            volumes_caps = [ {'name': 'ad-archive',
                          'hostPath':
                          {'path': adi_directory + '/volume/ad/archive'} } ]
        elif node_name == "ad-insertion-frontend":
            volumes_caps = [ {'name': 'ad-dash',
                          'hostPath':
                          {'path': adi_directory + '/volume/ad/dash'} },
                         {'name': 'ad-hls',
                          'hostPath':
                          {'path': adi_directory + '/volume/ad/hls'} } ]
        elif node_name == "content-provider":
            volumes_caps = [ {'name': 'html-volume',
                          'hostPath':
                          {'path': adi_directory + '/volume/html'} },
                         {'name': 'video-archive',
                          'hostPath':
                          {'path': adi_directory + '/volume/video/archive'} },
                         {'name': 'video-dash',
                          'hostPath':
                          {'path': adi_directory + '/volume/video/dash'} },
                         {'name': 'video-hls',
                          'hostPath':
                          {'path': adi_directory + '/volume/video/hls'} } ]
        elif node_name == "content-provider-transcode":
            volumes_caps = [ {'name': 'video-archive',
                          'hostPath':
                          {'path': adi_directory + '/volume/video/archive'} },
                         {'name': 'video-dash',
                          'hostPath':
                          {'path': adi_directory + '/volume/video/dash'} },
                         {'name': 'video-hls',
                          'hostPath':
                          {'path': adi_directory + '/volume/video/hls'} } ]
        elif node_name == "ad-transcode":
            volumes_caps = [ {'name': 'ad-dash',
                          'hostPath':
                          {'path': adi_directory + '/volume/ad/dash'} },
                         {'name': 'ad-hls',
                          'hostPath':
                          {'path': adi_directory + '/volume/ad/hls'} },
                         {'name': 'ad-static',
                          'hostPath':
                          {'path': adi_directory + '/volume/ad/static'} } ]
        elif node_name == "cdn":
            volumes_caps = [ {'name': 'secrets',
                          'secret': {'secretName': 'ssl-key-secret'} } ]

    data['spec']['template']['spec'].update({'volumes' : volumes_caps})
    with open(fileName, "w", encoding="utf-8") as outfile:
        yaml.dump(data, outfile, Dumper=yaml.RoundTripDumper, default_flow_style=False, allow_unicode=True)

def set_nodePort(data, fileName, port):
    data['spec']['ports'][0].update({'nodePort' : port})
    with open(fileName, "w", encoding="utf-8") as outfile:
        yaml.dump(data, outfile, Dumper=yaml.RoundTripDumper, default_flow_style=False, allow_unicode=True)
