/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>SIFBT Node</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.SIFBTNode#getName <em>Name</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.SIFBTNode#getChecks <em>Checks</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.SIFBTNode#getNodes <em>Nodes</em>}</li>
 * </ul>
 *
 * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getSIFBTNode()
 * @model
 * @generated
 */
public interface SIFBTNode extends BTreeNode
{
  /**
   * Returns the value of the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Name</em>' attribute.
   * @see #setName(String)
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getSIFBTNode_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link edu.vanderbilt.isis.alc.btree.bTree.SIFBTNode#getName <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Name</em>' attribute.
   * @see #getName()
   * @generated
   */
  void setName(String value);

  /**
   * Returns the value of the '<em><b>Checks</b></em>' reference list.
   * The list contents are of type {@link edu.vanderbilt.isis.alc.btree.bTree.CheckNode}.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Checks</em>' reference list.
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getSIFBTNode_Checks()
   * @model
   * @generated
   */
  EList<CheckNode> getChecks();

  /**
   * Returns the value of the '<em><b>Nodes</b></em>' containment reference list.
   * The list contents are of type {@link edu.vanderbilt.isis.alc.btree.bTree.ChildNode}.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Nodes</em>' containment reference list.
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getSIFBTNode_Nodes()
   * @model containment="true"
   * @generated
   */
  EList<ChildNode> getNodes();

} // SIFBTNode