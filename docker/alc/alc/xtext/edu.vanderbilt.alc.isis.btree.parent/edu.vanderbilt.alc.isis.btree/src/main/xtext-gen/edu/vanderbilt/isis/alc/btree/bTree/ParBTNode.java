/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Par BT Node</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.ParBTNode#getName <em>Name</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.ParBTNode#getCond <em>Cond</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.ParBTNode#getNodes <em>Nodes</em>}</li>
 * </ul>
 *
 * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getParBTNode()
 * @model
 * @generated
 */
public interface ParBTNode extends BTreeNode
{
  /**
   * Returns the value of the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Name</em>' attribute.
   * @see #setName(String)
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getParBTNode_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link edu.vanderbilt.isis.alc.btree.bTree.ParBTNode#getName <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Name</em>' attribute.
   * @see #getName()
   * @generated
   */
  void setName(String value);

  /**
   * Returns the value of the '<em><b>Cond</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Cond</em>' attribute.
   * @see #setCond(String)
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getParBTNode_Cond()
   * @model
   * @generated
   */
  String getCond();

  /**
   * Sets the value of the '{@link edu.vanderbilt.isis.alc.btree.bTree.ParBTNode#getCond <em>Cond</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Cond</em>' attribute.
   * @see #getCond()
   * @generated
   */
  void setCond(String value);

  /**
   * Returns the value of the '<em><b>Nodes</b></em>' containment reference list.
   * The list contents are of type {@link edu.vanderbilt.isis.alc.btree.bTree.ChildNode}.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Nodes</em>' containment reference list.
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getParBTNode_Nodes()
   * @model containment="true"
   * @generated
   */
  EList<ChildNode> getNodes();

} // ParBTNode
