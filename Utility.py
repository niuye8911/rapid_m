def RAPID_warn(prefix, message):
    print "RAPID_LEARNER WARNING: " + str(prefix) + ":" + str(message)


def RAPID_info(prefix, message):
    print "RAPID_LEARNER INFO: " + str(prefix) + ":" + str(message)


def not_none(values):
    return reduce(lambda x, y: x and y, values, True)
