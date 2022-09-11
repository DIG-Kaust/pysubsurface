"""
Additional colormaps for pysubsurface API
"""
from matplotlib.colors import LinearSegmentedColormap

# Define custom colormaps
cmap_avoexplor = \
    LinearSegmentedColormap.from_list('name', ['red', 'orange', 'yellow',
                                               'white', '#4da6ff', 'blue',
                                               'black'])

cmap_correxplor = \
    LinearSegmentedColormap.from_list('name', ['black', 'black', 'black',
                                               'black', 'black', 'purple',
                                               'red', 'yellow', 'white'])

cmap_corrpetrel = \
    LinearSegmentedColormap.from_list('name', ['#66ffff', '#33ccff', 'blue',
                                               '#404040', 'grey', '#404040',
                                               'red', '#ffff00', '#ffff00'])

cmap_corrpetrel_white = \
    LinearSegmentedColormap.from_list('name', ['#66ffff', '#33ccff', 'blue',
                                               '#404040', 'white', '#404040',
                                               'red', '#ffff00', '#ffff00'])

cmap_yrbwpetrel = \
    LinearSegmentedColormap.from_list('name', ['yellow', 'red', 'black',
                                               'grey', 'white'])


cmap_amplitudepkdsg = \
    LinearSegmentedColormap.from_list('name', ['#33ffff', '#33adff', '#0000ff',
                                               '#666666', '#d9d9d9', '#805500',
                                               '#ff6600', '#ffdb4d', '#ffff00'])
cmap_amplitudepkdsg_r = \
    LinearSegmentedColormap.from_list('name',
                                      list(reversed(['#33ffff', '#33adff',
                                                     '#0000ff', '#666666',
                                                     '#d9d9d9', '#805500',
                                                     '#ff6600', '#ffdb4d',
                                                     '#ffff00'])))

cmap_hordsg = LinearSegmentedColormap.from_list('name', ['#800000', '#ffdb4d',
                                                         '#006600', '#0099cc',
                                                         '#0059b3'])

cmap_hordsg_r = LinearSegmentedColormap.from_list('name',
                                                  list(reversed(['#800000', '#ffdb4d',
                                                                 '#006600', '#0099cc',
                                                                 '#0059b3'])))

cmap_interleavedsg = LinearSegmentedColormap.from_list('name', ['#ffffff',
                                                                '#404040',
                                                                '#808080',
                                                                '#bfbfbf',
                                                                '#000000',
                                                                '#0099ff',
                                                                '#40ff00',
                                                                '#ffff00',
                                                                '#ff0000'])

cmap_interleavedsg_r = LinearSegmentedColormap.from_list('name',
                                                         list(reversed(['#ffffff',
                                                                        '#404040',
                                                                        '#808080',
                                                                        '#bfbfbf',
                                                                        '#000000',
                                                                        '#0099ff',
                                                                        '#40ff00',
                                                                        '#ffff00',
                                                                        '#ff0000'])))

cmap_bluorange = LinearSegmentedColormap.from_list('name',
                                                   list(reversed(['#004d99',
                                                                  '#ffffff',
                                                                  '#ff6600'])))

cmap_bluorange_r = LinearSegmentedColormap.from_list('name',
                                                   list(reversed(['#ff6600',
                                                                  '#ffffff',
                                                                  '#004d99'])))


cmap_shale = LinearSegmentedColormap.from_list('name', ['#ffff00', '#444422'])


cmaps = {'avoexplor': cmap_avoexplor,
         'correxplor': cmap_correxplor,
         'corrpetrel': cmap_corrpetrel,
         'corrpetrel_white': cmap_corrpetrel_white,
         'yrbwpetrel': cmap_yrbwpetrel,
         'cmap_amplitudepkdsg': cmap_amplitudepkdsg,
         'cmap_amplitudepkdsg_r': cmap_amplitudepkdsg_r,
         'cmap_hordsg': cmap_hordsg,
         'cmap_hordsg_r': cmap_hordsg_r,
         'cmap_interleavedsg': cmap_interleavedsg,
         'cmap_interleavedsg_r': cmap_interleavedsg_r,
         'cmap_bluorange': cmap_bluorange,
         'cmap_bluorange_r': cmap_bluorange_r,
         'cmap_shale': cmap_shale}
