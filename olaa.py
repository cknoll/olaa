import pyirk as p

__URI__ = "irk:/olaa/0.1"
keymanager = p.KeyManager()
p.register_mod(__URI__, keymanager)
p.start_mod(__URI__)

#TOP-LEVEL CONCEPT
I2000 = p.create_item(
    "I2000",
    R1__has_label="Linear Algebra",
    R2__has_description=
        "A branch of mathematics concerned with vector spaces, linear transformations, "
        "and systems of linear equations. It serves as a foundation for numerous fields "
        "such as machine learning, computer graphics, and physics."
    
)
I2001 = p.create_item(
    "I2001",
    R1__has_label="Eigenvector",
    R2__has_description=
        "a non-zero vector that, when a linear transformation (represented by a matrix)"
        "is applied to it, only gets scaled by a scalar value (the eigenvalue) without changing direction."
        
)

I2002 = p.create_item(
    "I2002",
    R1__has_label="Eigenvalue",
    R2__has_description=
        " a scalar that represents how much an eigenvector is stretched or compressed during a linear transformation."
        
)

# KEYWORDS IN LINEAR ALGEBRA RELATED TO EIGEN VECTORS AND VALUES

# Linear Transformation
I2003 = p.create_item(
    "I2003",
    R1__has_label="Linear Transformation",
    R2__has_description=
        "A linear transformation is a mathematical function that maps vectors to other vectors," 
        "preserving vector addition and scalar multiplication. It can be represented by a matrix acting on a vector."
    
)

# Geometrical Vectors
I2004 = p.create_item(
    "I2004",
    R1__has_label="Geometrical Vector",
    R2__has_description=
        "A quantities that has both magnitude and direction. It is used to represent points, directions,"
        "or forces in space, and can be added or scaled by a scalar"
)

# Square Matrices
I2005 = p.create_item(
    "I2005",
    R1__has_label="Square Matrix",
    R2__has_description=
        "a matrix that has the same number of rows and columns. It is often used in linear "
        "transformations and solving systems of linear equations."
    
)
    
#Eigen function
I2006 = p.create_item(
    "I2006",
    R1__has_label="Eigen function",
    R2__has_description=
        "functions that, when a linear operator is applied to them, result in the same function "
        "scaled by a constant (similar to eigenvectors for matrices). They arise in contexts like"
        " differential equations and quantum mechanics."
    
)

##################################################################################################################################

# MAJOR SUBFIELDS

#MATRIX DIAGONALIZATION
I2007 = p.create_item(
    "I2007",
    R1__has_label="Matrix Diagonalization",
    R2__has_description=
        "The process of transforming a square matrix into a diagonal matrix using its eigenvalues "
        "and eigenvectors, which simplifies computations such as matrix exponentiation."
    
)

#SPECTRAL DECOMPOSITION
I2008 = p.create_item(
    "I2008",
    R1__has_label="Spectral Decomposition",
    R2__has_description=
        "the process of decomposing a matrix (usually symmetric) into its eigenvalues and eigenvectors. " 
        "It expresses the matrix as a sum of outer products of eigenvectors weighted by their corresponding eigenvalues."

    
)
#PCA
I2009 = p.create_item(
    "I2009",
    R1__has_label="Principal Component Analysis (PCA)",
    R2__has_description=
        "A dimensionality reduction technique that identifies the directions (principal components) "
        "in data where variance is maximized, based on eigenvectors and eigenvalues of the covariance matrix."
    
)
#SVD
I2010 = p.create_item(
    "I2010",
    R1__has_label="Singular Value Decomposition (SVD)",
    R2__has_description=
        "A factorization of a matrix into three matrices, revealing its rank, range, and null space. "
        "It generalizes the concept of eigenvalues and eigenvectors for all matrices."
    
)

#Fundamental theorem of algebra
I2011 = p.create_item(
    "I2011",
    R1__has_label="Fundamental theorem of algebra",
    R2__has_description=
        "It states that every non-constant polynomial equation with complex coefficients has at least one "
        "complex root. In other words, any polynomial can be factored into linear factors over the complex numbers."
    
)

##################################################################################################################################

# COMMON ALGORITHMS INVOLVING EIGENVALUES

#Power Iteration
I2012 = p.create_item(
    "I2012",
    R1__has_label="Power Iteration",
    R2__has_description=
        "An iterative algorithm to approximate the dominant eigenvalue and its corresponding eigenvector "
        "of a matrix. Used in applications like PageRank."
)
#Inverse Iteration
I2013 = p.create_item(
    "I2013",
    R1__has_label="Inverse Iteration",
    R2__has_description=
        "An algorithm to approximate an eigenvector corresponding to a given eigenvalue of a matrix. "
        "Effective for finding eigenvectors near a specific eigenvalue."
    
)
#Eigenvalue Stability Analysis
I2014 = p.create_item(
    "I2014",
    R1__has_label="Eigenvalue Stability Analysis",
    R2__has_description=
        "A method used in control theory and dynamical systems to determine the stability of a system "
        "based on the eigenvalues of its state matrix."
    
)

I2015 = p.create_item(
    "I2014",
    R1__has_label="Dimensionality Reduction",
    R2__has_description=
        "process of reducing the number of features or variables in a dataset while preserving its"
        " essential structure and information. It helps simplify data analysis, reduce computation"
        " costs, and mitigate issues like overfitting."   
)

I2016 = p.create_item(
    "I2016",
    R1__has_label="Matrix",
    R2__has_description=
        "A matrix is a structured, rectangular arrangement of elements organized in rows and columns, "
        "used to represent data or solve mathematical problems." 
)
# RELATIONS

# Matrix as fundamental part of Linear Algebra, square matrix a subclass of Matrix which further relates to Eigenvalue and Eigenvector
# NOTE: Eigenvalue and eigenvectors are indirectly related to Linear Algebra through Matrix and square matrix

I2016["Matrix"].set_relation(p.R6["has defining mathematical relation"], I2000["Linear Algebra"])
I2005["Square Matrix"].set_relation(p.R3["is a subclass of"], I2016["Matrix"]) 
I2005["Square Matrix"].set_relation(p.R31["is in mathematical relation with"], I2001["Eigenvector"])
I2005["Square Matrix"].set_relation(p.R31["is in mathematical relation with"], I2002["Eigenvalue"]) 
I2002["Eigenvalue"].set_relation(p.R6["has defining mathematical relation"], I2001["Eigenvector"])

#Setting up relations for matrix diagonalisation
I2007["Matrix Diagonalization"].set_relation(p.R6["has defining mathematical relation"], I2001["Eigenvector"]) 
I2007["Matrix Diagonalization"].set_relation(p.R3["is subclass of"], I2008["Spectral Decomposition"]) 

# Algorithms as Instances of Eigenvalue Problems
I2012["Power Iteration"].set_relation(p.R12["is defined by means of"], I2002["Eigenvalue"]) 
I2013["Inverse Iteration"].set_relation(p.R12["is defined by means of"], I2001["Eigenvector"]) 
I2014["Eigenvalue Stability Analysis"].set_relation(p.R12["is defined by means of"], I2002["Eigenvalue"]) 
I2014["Eigenvalue Stability Analysis"].set_relation(p.R5["is part of"], I2000["Linear Algebra"]) 

#
I2009["Principal Component Analysis (PCA)"].set_relation(p.R12["is defined by means of"], I2007["Matrix Diagonalization"])  
I2004["Geometrical Vector"].set_relation(p.R5["is part of"], I2000["Linear Algebra"]) 
I2004["Geometrical Vector"].set_relation(p.R31["is in mathematical relation with"], I2001["Eigenvector"])  

# PCA and SVD as instances of dimensionality reduction
I2009["Principal Component Analysis (PCA)"].set_relation(p.R4["is instance of"], I2015["Dimensionality Reduction"])  
I2010["Singular Value Decomposition (SVD)"].set_relation(p.R4["is instance of"], I2015["Dimensionality Reduction"])  
I2009["Principal Component Analysis (PCA)"].set_relation(p.R31["is in mathematical relation with"], I2010["Singular Value Decomposition (SVD)"])    

# Mathematically Relating the Fundamental Theorem of Algebra to Eigenvalues and Eigenvectors 
I2011["Fundamental theorem of algebra"].set_relation(p.R31["is in mathematical relation with"], I2001["Eigenvalue"])
I2007["Matrix Diagonalization"].set_relation(p.R12["is defined by means of"], I2002["Eigenvalue"])  

# Linear Transformation as a foundation for Eigenvalues and Eigenvectors
I2003["Linear Transformation"].set_relation(p.R5["is part of"], I2000["Linear Algebra"])
I2003["Linear Transformation"].set_relation(p.R31["is in mathematical relation with"], I2001["Eigenvector"]) 
I2003["Linear Transformation"].set_relation(p.R5["is related to"], I2000["Linear Algebra"])

# Eigen functions
I2006["Eigen function"].set_relation(p.R5["is part of"], I2000["Linear Algebra"])
I2006["Eigen function"].set_relation(p.R31["is in mathematical relation with"], I2002["Eigenvalue"])

p.end_mod()
