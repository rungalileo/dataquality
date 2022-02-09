from enum import Enum


class Route(str, Enum):
    """
    List of available API routes
    """

    proc_pool = "proc/pool"
    projects = "projects"
    runs = "runs"
    users = "users"
    cleanup = "cleanup"
    login = "login"
    current_user = "current_user"
    healthcheck = "healthcheck"
    proc = "proc"
    slices = "slices"
