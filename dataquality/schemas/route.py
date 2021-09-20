from enum import Enum


class Route(str, Enum):
    """
    List of available API routes
    """
    pipelines = 'pipelines'
    projects = 'projects'
    users = 'users'
    cleanup = 'cleanup'
