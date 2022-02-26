import pstats

path = '.\\volume\cloud\stats\POST.endpoint.1467ms.1645789039.prof'
p = pstats.Stats(
    path)
p.sort_stats('cumulative').print_stats('helper_cloud', 10)
