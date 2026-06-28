import os

def validate_and_normalize_path(path: str, must_be_in_workspace: bool = False) -> str:
    """
    Validates and normalizes paths to prevent path traversal outside the repository.
    If must_be_in_workspace is True, restricts access specifically to the workspace directory.
    
    Args:
        path: The file path to validate.
        must_be_in_workspace: If True, forces path to stay within workspace/.
        
    Returns:
        The normalized path.
        
    Raises:
        ValueError: If path traversal or unauthorized absolute path is detected when must_be_in_workspace is False.
    """
    clean_path = os.path.normpath(path)
    
    # Prevent traversal or absolute paths breaking repository boundaries
    if clean_path.startswith("..") or os.path.isabs(clean_path):
        if must_be_in_workspace:
            # Fallback to putting the file directly in workspace
            base = os.path.basename(clean_path)
            return os.path.join("workspace", base)
        raise ValueError(f"Access denied to path '{path}'. Paths must be relative and stay within the repository.")
        
    if must_be_in_workspace:
        if not (clean_path.startswith("workspace") or clean_path.startswith("./workspace")):
            clean_path = os.path.join("workspace", clean_path)
            
    return clean_path
