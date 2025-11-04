# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:58:25 2022

@author: mpanze
"""
from pathlib import Path
import login
from datajoint.user_tables import Table
from typing import List, Union

def normalise_path(p: Union[str,Path]) -> Path:
    """
    Returns a path object with cross-platform compatibility
    """
    return Path(str(p).replace("\\","/"))

def load_data_directories() -> List[Path]:
    """
    Loads main + alternative data directories specified by the login.py file as Path objects

    Returns
    -------
    List[Path]
        List of possible data directories as pathlib.Path objects.
    """
    try:    # try loading alternative dirs
        roots = [login.get_neurophys_data_directory(), *login.get_alternative_data_directories()]
    except AttributeError:  # if not defined by user, load only neurophys drive
        roots = [login.get_neurophys_data_directory()]
    # replace pesky backslashes with forward slashes and convert to Paths
    roots = [normalise_path(r) for r in roots]
    return roots

def get_relative_path(absolute_path: Union[str, Path], subfolder: Union[str, Path] = None) -> Path:
    """
    Computes path relative to (any valid) data directory.

    Parameters
    ----------
    absolute_path : Union[str, Path]
        input path to be transformed to a relative path.
    subfolder : Union[str, Path], optional
        specify a directory of the base data directory for computing the relative path. The default is None.

    Returns
    -------
    Path
        relative path as a pathlib.Path object.
    """
    # get root directories
    roots = load_data_directories()
    if subfolder:
        roots = [r / normalise_path(subfolder) for r in roots]
        
    # normalise input path
    abs_path = normalise_path(absolute_path)
    
    # iterate over paths
    valid_paths = [abs_path.relative_to(r)
                   for r in roots
                   if r in abs_path.parents]
    if len(valid_paths) == 0:
        raise ImportError("{} cannot form a relative path to any of the following DATA directories:\{}"
                          .format(abs_path, roots))
    return valid_paths[0]

def unique(paths: List[Path]) -> Path:
    """
    Return subset of paths which have unique relative paths

    Parameters
    ----------
    paths : List[Path]
        input paths.

    Returns
    -------
    Path
        output list of unique paths.
    """
    rel_paths = []
    unique_paths = []
    for p in paths:
        p_rel = get_relative_path(p)
        if not p_rel in rel_paths:
            rel_paths.append(p_rel)
            unique_paths.append(p)
    return unique_paths

def glob(pattern: str, recursive: bool = False, subfolder: Union[str, Path] = None) -> List[Path]:
    """
    Finds all unique files that match the specified pattern, relative to all possible data directories.
    If one wants to match patterns relative to a specific subfolder, e.g. a session folder,
    the 'subfolder' parameter can be specified.

    Parameters
    ----------
    pattern : str
        pattern to match, standard globbing patterns and wildcards are accepted.
    recursive : bool, optional
        if True, will also search in subfolders. The default is False.
    subfolder : Union[str, Path], optional
        specify a subdirectory of the base data directory. The default is None.

    Returns
    -------
    List[Path]
        List of absolute paths.
    """
    # get root directories
    roots = load_data_directories()
    if subfolder:
        roots = [r / normalise_path(subfolder) for r in roots]
    
    # glob everything
    paths = []
    for r in roots:
        if recursive:
            paths += list(r.rglob(pattern))
        else:
            paths += list(r.glob(pattern))
    
    # eliminate duplicates
    return unique(paths)

def get_absolute_paths(query: Table, attribute: str) -> List[Path]:
    """
    Returns absolute location of files in a table

    Parameters
    ----------
    query : Table
        Table which stores the relative file paths. Pass the table to compute absolute paths for all entries
        or query objects deriving from the table (e.g. table & restriction).
        Must depend either directly or indirectly on common_exp.Session table.
    attribute : str
        Attribute name of the file path within the query.

    Raises
    ------
    FileNotFoundError
        If the file cannot be located in any user-defined storage location.

    Returns
    -------
    List[Path]
        Absolute paths of the files in the query as pathlib.Path objects.
    """
    from schema.common_exp import Session
    sessions = (Session.proj("session_path") * query.proj(file_path = attribute))
    
    roots = load_data_directories()
    
    abs_paths = []
    for sess in sessions:
        # replace backslashes for cross-platform compatibility
        s_path = normalise_path(sess["session_path"])
        f_path = normalise_path(sess["file_path"])
        # compute vaild paths
        valid_paths = [r / s_path / f_path
                       for r in roots
                       if (r/s_path/f_path).exists()]
        # check if any files are found
        if len(valid_paths) == 0:
            raise FileNotFoundError("File {} from {} was not found in any of the following directories: \n{}".format(f_path,s_path,roots))
        abs_paths += unique(valid_paths)
    return abs_paths