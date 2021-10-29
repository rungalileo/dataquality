from enum import Enum


class Route(str, Enum):
    """
    List of available API routes
    """

    proc_pool = "proc/pool"
    projects = "projects"
    users = "users"
    cleanup = "cleanup"
