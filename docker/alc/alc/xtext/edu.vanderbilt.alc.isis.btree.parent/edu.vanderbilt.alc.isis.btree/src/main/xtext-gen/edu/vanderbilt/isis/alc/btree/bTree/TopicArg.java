/**
 * generated by Xtext 2.25.0
 */
package edu.vanderbilt.isis.alc.btree.bTree;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Topic Arg</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.TopicArg#getType <em>Type</em>}</li>
 *   <li>{@link edu.vanderbilt.isis.alc.btree.bTree.TopicArg#getName <em>Name</em>}</li>
 * </ul>
 *
 * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getTopicArg()
 * @model
 * @generated
 */
public interface TopicArg extends EObject
{
  /**
   * Returns the value of the '<em><b>Type</b></em>' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Type</em>' reference.
   * @see #setType(Topic)
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getTopicArg_Type()
   * @model
   * @generated
   */
  Topic getType();

  /**
   * Sets the value of the '{@link edu.vanderbilt.isis.alc.btree.bTree.TopicArg#getType <em>Type</em>}' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Type</em>' reference.
   * @see #getType()
   * @generated
   */
  void setType(Topic value);

  /**
   * Returns the value of the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the value of the '<em>Name</em>' attribute.
   * @see #setName(String)
   * @see edu.vanderbilt.isis.alc.btree.bTree.BTreePackage#getTopicArg_Name()
   * @model
   * @generated
   */
  String getName();

  /**
   * Sets the value of the '{@link edu.vanderbilt.isis.alc.btree.bTree.TopicArg#getName <em>Name</em>}' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @param value the new value of the '<em>Name</em>' attribute.
   * @see #getName()
   * @generated
   */
  void setName(String value);

} // TopicArg
